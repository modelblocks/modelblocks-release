#
# repairToEdited.rb 
# alters output cnftrees to get rid of REPAIR constituents and 
# add in EDITED constituents instead, since that is how the gold
# standard is annotated

require 'scripts/umnlp.rb'

class Tree
	def addEdit!
		if @children == nil or @children.size == 0
			return
		end

		@children.each_index { |x|
			if @children[x].head.include?("REPAIR") 
				myindex = (@parent == nil ? 0 : (@parent.children == nil ? 0 : @parent.children.index(self)))
				#$stderr.puts "found one: #{@children[x].to_s} myindex is #{myindex.to_s}"
				if x > 0
					@children[x].head = @children[x].head.gsub("REPAIR","")
					@children[x-1].head = "EDITED"
				elsif myindex > 0 and @parent.children[myindex-1].head == @children[x].head.gsub("REPAIR","")
					# structure is
					#     X  
					#    / \ 
					#   Y   Z
					#       |  
					#    REPAIRY
					@parent.children[myindex-1].head = "EDITED"
					@children[x].head = @children[x].head.gsub("REPAIR","")
				else
					parentindex = 0
					if @parent.parent != nil and @parent.parent.children != nil
						parentindex = @parent.parent.children.index(@parent)
					end
					if parentindex > 0 and @parent.parent.children[parentindex-1].head == @children[x].head.gsub("REPAIR","")
						# structure is
						#        W
						#       / \
						#      Y   X
						#          |
						#          Z
						#          |
						#       REPAIRY
						@parent.parent.children[parentindex-1].head = "EDITED"
						@children[x].head = @children[x].head.gsub("REPAIR","")
					else
						# catch-all case... just trash the repair
						@children[x].head = @children[x].head.gsub("REPAIR","")
					end
				end
			end

			@children[x].addEdit!
		}
	end
end

while(line = gets)
	t = Tree.new(line)
	t.addEdit!
	puts t.to_s
end
